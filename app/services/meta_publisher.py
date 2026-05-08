from typing import Any

import requests

from app.config import get_settings


class MetaPublishError(RuntimeError):
    pass


def _raise_for_meta(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {'text': response.text}

    if response.status_code >= 400 or ('error' in payload):
        raise MetaPublishError(f'Meta API error: {payload}')
    return payload


def _graph_url(path: str) -> str:
    settings = get_settings()
    version = settings.meta_graph_version.strip('/')
    return f'https://graph.facebook.com/{version}/{path.lstrip("/")}'


def publish_facebook_post(message: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.facebook_page_id or not settings.facebook_page_access_token:
        raise RuntimeError('FACEBOOK_PAGE_ID and FACEBOOK_PAGE_ACCESS_TOKEN are required for Facebook publishing.')

    url = _graph_url(f'{settings.facebook_page_id}/feed')
    response = requests.post(
        url,
        data={
            'message': message,
            'access_token': settings.facebook_page_access_token,
        },
        timeout=30,
    )
    return _raise_for_meta(response)


def create_instagram_media_container(image_url: str, caption: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.instagram_user_id or not settings.instagram_access_token:
        raise RuntimeError('INSTAGRAM_USER_ID and INSTAGRAM_ACCESS_TOKEN are required for Instagram publishing.')

    url = _graph_url(f'{settings.instagram_user_id}/media')
    response = requests.post(
        url,
        data={
            'image_url': image_url,
            'caption': caption,
            'access_token': settings.instagram_access_token,
        },
        timeout=30,
    )
    return _raise_for_meta(response)


def publish_instagram_container(creation_id: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.instagram_user_id or not settings.instagram_access_token:
        raise RuntimeError('INSTAGRAM_USER_ID and INSTAGRAM_ACCESS_TOKEN are required for Instagram publishing.')

    url = _graph_url(f'{settings.instagram_user_id}/media_publish')
    response = requests.post(
        url,
        data={
            'creation_id': creation_id,
            'access_token': settings.instagram_access_token,
        },
        timeout=30,
    )
    return _raise_for_meta(response)


def publish_instagram_image(image_url: str, caption: str) -> dict[str, Any]:
    container = create_instagram_media_container(image_url=image_url, caption=caption)
    creation_id = container.get('id')
    if not creation_id:
        raise MetaPublishError(f'Instagram media container did not return an id: {container}')
    published = publish_instagram_container(creation_id)
    return {'container': container, 'published': published}
